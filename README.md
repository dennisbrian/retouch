# Pro Max Face Retouch Engine

[![Tests](https://github.com/dennisbrian/retouch/actions/workflows/test.yml/badge.svg)](https://github.com/dennisbrian/retouch/actions/workflows/test.yml)
[![Benchmarks](https://github.com/dennisbrian/retouch/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/dennisbrian/retouch/actions/workflows/benchmarks.yml)

Professional automated face retouching for portraits, cosplay, and batch workflows. Combines MediaPipe landmarks, BiSeNet semantic segmentation, frequency-separation skin work, and global color grading.

## Requirements

- Python 3.9–3.11 (3.11 recommended). MediaPipe 0.10.5 has no wheels for 3.12+.
- macOS, Linux, or Windows

## Quick setup (macOS and Linux)

```bash
./setup        # installs uv if needed, builds .venv from uv.lock, fetches the core models
./run          # opens the app: a native window on macOS, your browser elsewhere
```

`./setup` is safe to re-run. `.python-version` pins Python 3.11, and uv
downloads that interpreter if your system does not have it. Other launcher
commands:

```bash
./run web                                  # open the app in your browser
./run batch ~/photos -o ~/photos_out --recipe natural --workers 4
./run batch ~/dim_shots -o ~/out --recipe con_high_iso_v1  # high-ISO: AI denoise first
./run crops ~/photos_out --formats 4:5,9:16,1:1  # export face-aware crops for Instagram, TikTok, etc.
./run review apply ~/photos_out decisions.json  # copy the picks you marked in review.html
./run recipes                              # list the recipe names --recipe accepts
./run lut cosplay_feed_pop_v1 -o ~/luts    # save a recipe's colour look as a .cube LUT
./run watermark ~/photos_out --text "© Alex Studio {year}"  # credit on copies for posting
./run update                               # git pull, then refresh dependencies
```

## Manual install

On Windows, or if you prefer pip:

```bash
python -m venv .venv          # use Python 3.9–3.11
pip install -r requirements/base.txt
```

Optional extras:

```bash
pip install -r requirements/dev.txt   # pytest
pip install -r requirements/gui.txt   # Gradio web UI
pip install -r requirements/raw.txt   # RAW camera file support
```

RetinaFace is not supported: its dependencies conflict with the pinned
MediaPipe runtime, so MediaPipe is the only detection path.

## Model files

The three core MediaPipe models (about 13 MB together) download automatically,
verified against `models/manifest.json`, into `~/.cache/retouch/models` the
first time they are needed; `./setup` fetches them up front. Set
`RETOUCH_CACHE_DIR` to use another location. To place them in `models/`
by hand instead (for example on an offline machine):

```bash
mkdir -p models

# MediaPipe face landmarker (downloaded on demand; verify before use)
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task?generation=1683136941916318"

# MediaPipe selfie segmenter (downloaded on demand for hair/person masking)
curl -L -o models/selfie_segmenter.tflite \
  "https://storage.googleapis.com/download/storage/v1/b/mediapipe-models/o/image_segmenter%2Fselfie_segmenter%2Ffloat16%2Flatest%2Fselfie_segmenter.tflite?alt=media&generation=1683436453600523"

# MediaPipe multiclass selfie segmenter (downloaded on demand for hair/neck
# masks when BiSeNet is not installed; Apache-2.0)
curl -L -o models/selfie_multiclass_256x256.tflite \
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite?generation=1682480017063560"

# MediaPipe Pose Landmarker Full (downloaded on demand for body reshape)
curl -L -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task?generation=1682642785209422"

# NAFNet AI denoise (117 MB, MIT; downloaded on demand the first time
# AI denoise is used; hosted on this repo's models-nafnet-v1 pre-release)
curl -L -o models/nafnet_denoise.onnx \
  "https://github.com/dennisbrian/retouch/releases/download/models-nafnet-v1/nafnet_denoise.onnx"

# Verify every downloaded artifact against models/manifest.json.
python scripts/qa/verify_models.py \
  --model face_landmarker --model selfie_segmenter --model selfie_multiclass \
  --model pose_landmarker_full --model denoise_nafnet

# Optional local BiSeNet face parsing ONNX (not distributed by the wheel)
# Place a manifest-matching resnet18.onnx export at:
#   models/resnet18.onnx
```

The engine falls back to landmark-based masks if `resnet18.onnx` is missing.
The manifest reports this model as unavailable until a verified release URL is
provided; local binaries are accepted only when their SHA-256 and size match.
BiSeNet's weights are trained on CelebAMask-HQ, which is non-commercial, so
they are not shipped. On that fallback path the MediaPipe multiclass segmenter
supplies the hair and neck masks and cuts bangs and accessories out of the skin
mask. Without it (offline, or `RETOUCH_CLASS_SEGMENTER=0`) hair is a band of
the person mask around the head and there is no neck mask.

AI denoise (`--ai-denoise 0-100` in the CLI, the "AI Noise Reduction" slider
under Tone & Light in the app, or recipes such as `con_high_iso_v1`) runs the
NAFNet model before retouching. It is CPU-only and takes roughly 10 s per
1.5 MP (about 80 s at 24 MP), so keep it for noisy high-ISO shots. If the model
cannot be downloaded, a bilateral filter is used instead and the result is
flagged for review.

Glasses glare removal (`--lens-glare 0-100` in the CLI, the "Glasses Glare
Removal" slider under Eyes & Lips in the app, or `glasses.deglare` in a recipe)
lifts flash, softbox and window reflections off glasses, goggles and visors over
the eyes. It is off by default and uses no model. It cannot recover detail that
the glare blew out completely (those spots are refilled with the surrounding
tone), leaves a reflection that sits only inside the eye opening alone (it looks
like a catchlight), and does not reach glare below the eyes on a full visor. On
faces without glasses it also softens shine around the eyes, so use it on
photos with eyewear.

Set `RETOUCH_OFFLINE=1` before launching the GUI to disable update checks and
prevent model downloads; the Advanced Retouch status panel shows the mode.

## Quick start

### Python API

```python
import cv2
from retouch import RetouchEngine

img = cv2.imread("portrait.jpg")
with RetouchEngine() as engine:
    result = engine.process(img, recipe="natural", smooth=50, whiten=20)
cv2.imwrite("portrait_retouched.jpg", result)
```

### Batch CLI

```bash
python3 cli.py /path/to/photos -o /path/to/output --recipe cosplay_clear_v1 --workers 4
```

To keep a set consistent for a carousel, add `--match-hero path/to/best.jpg`:
every photo's exposure and white balance is matched to that hero frame on the
subject's skin before the recipe runs (see "Match a Set to One Hero Frame" in
[BATCH_GUIDE.md](docs/guides/BATCH_GUIDE.md)).

Each batch also writes `review.html` into the output folder: open it in a browser to check before/after, face close-ups and QA flags, mark picks and rejects from the keyboard, then export the decisions and apply them to copy the picks into a folder. See [BATCH_GUIDE.md](docs/guides/BATCH_GUIDE.md) for full CLI options and the review workflow.

### Edit reports and Content Credentials

```bash
./run batch shoot/ -o out/ --recipe natural --edit-report
./run batch shoot/ -o out/ --recipe natural --sign-cert chain.pem --sign-key signing.key
```

`--edit-report` writes `out/edit-reports/<photo>.json` for each photo: the
recipe, every active edit grouped (skin, makeup, eyes, colour...), whether any
face or body shape was changed, AI use, and how much of the frame changed.
`--sign-cert`/`--sign-key` sign each retouched photo with Content Credentials
(C2PA) using your own certificate, with that report inside and the original
photo recorded as its parent (needs `uv sync --extra desktop --extra credentials`).
A camera's own Content Credentials are no longer copied onto retouched
photos, where they failed verification. See
[CONTENT_CREDENTIALS.md](docs/guides/CONTENT_CREDENTIALS.md).

### Split a shoot by capture time

```bash
./run split ~/shoots/2026-09-20            # preview the sets
./run split ~/shoots/2026-09-20 --move     # one folder per set; RAF+JPG pairs stay together
```

A new set starts after a 15-minute pause (`--gap`), or use `--by hour` / `--by day`.
See [SPLIT_SHOOT.md](docs/guides/SPLIT_SHOOT.md).

The CLI also accepts several literal input paths and can save a deterministic
selection/output plan before rendering:

```bash
python3 cli.py --dry-run --input-plan selection.json \
  "/path/to/A 01.jpg" "/path/to/B 02.RAF" \
  -o "/path/to/retouched"
```

### Recipe QA

Compare recipes on one photo or sweep selected recipes across a folder before
committing to a batch. The runners write contact sheets, comparison images, and
machine-readable manifests:

```bash
# One photo across selected recipes
./executable/recipe_sweep portrait.jpg --recipes natural,portrait,cosplay --compare

# Folder validation at full resolution: omit --max-dim
./executable/recipe_batch /path/to/photos -o test_output/recipe_check \
  --recipes cosplay_character_showcase_v1,cosplay_pastel_dream_showcase_v1 --compare
```

See [RECIPE_SWEEP.md](docs/RECIPE_SWEEP.md) for the full visual-QA workflow.

### Random image system test

Pick one JPEG at random from an image folder and run it through the retouch
pipeline. The output includes the selected source, recipe renders,
`manifest.json`, and a contact sheet:

```bash
./executable/random_image_test /path/to/my/images --count 5 \
  --recipes recommended --compare
```

Use `--seed 42` to repeat the same random selections. Each image gets its own
folder plus a `batch_manifest.json` under `test_output/random_image_tests/`.
For a quick local smoke test without face-model initialization, add
`--global-only`.

### Web GUI

```bash
./run web            # or, in a manual install: python3 gui.py
```

Opens at `http://127.0.0.1:7860`. Press Ctrl+C in the terminal to stop it,
and start it again after updating.

### Standalone app

`scripts/build/retouch_app.spec` builds a self-contained desktop app with
PyInstaller, so end users need no Python. Builds are per platform (macOS,
Windows, Linux); the `Desktop builds` workflow produces all three. See
[BUILD.md](BUILD.md).

## Recipes

The CLI `--recipe` flag and the GUI dropdowns take names from the curated
catalog (`CURATED_RECIPE_NAMES` in `retouch/recipes.py`), for example `natural`,
`portrait`, `natural_polish_v1`, `cosplay_clear_v1`,
`cosplay_character_showcase_v1`, `apex_cinema_v1` and `reala_ace`. List them all:

```bash
./run recipes
```

Cosplay looks for specific scenes (all under "Scene / creative" in the app):

| Recipe | Use it for |
|---|---|
| `cosplay_neon_night_v1` | Night shots under neon, LED signs or RGB stage light. Keeps the scene's colour, pulls skin back to its own tone, makes signs glow. |
| `cosplay_dark_villain_v1` | Villain, gothic and dark-fantasy characters. Low-key and cool, keeps dark lipstick, contour and drawn marks, no added blush. |
| `cosplay_feed_pop_v1` | Posts for Instagram, Reels and TikTok. Extra punch and subject sharpness that survive the app's compression, highlights held for white wigs. |

`python3 cli.py --list-recipes` shows every recipe in the cookbook, including
older looks such as `cosplay`, `anime_v2` and the Fuji film sims. Those are not
in the curated catalog, so `--recipe` rejects them with "invalid choice"; they
still work through the Python API, e.g. `engine.process(img, recipe="anime_v2")`.

### Body paint

For blue, green, grey or purple painted skin, add `--body-paint 0-100` to any
recipe (or use the "Body Paint" slider under Cosplay & Body in the app). It is
off by default. When it is on and a face's skin reads as paint, the paint keeps
its own colour through skin retouching (without it, grey paint picks up a pink
cast and blue or green paint drifts a few degrees toward skin tones), and the
strength evens out patchy, thin or sponge-marked coverage, including warm skin
showing through a thin coat. Painted neck, shoulders and arms that join the
face in the same colour are included. Colour grading still applies on top.

```bash
python3 cli.py shoot/ -o out/ --recipe cosplay_clear_v1 --body-paint 60
```

Red, orange and tan paint look like skin, so they are not detected. A costume
in exactly the paint's colour that touches painted skin is evened too. On a
black-and-white photo grey skin is not treated as paint. It added about 10 s on
a 45 MP test frame, and nothing on photos without paint. Details are in
`retouch/body_paint.py`.

### Recipe gallery

Not sure which recipe suits a photo? In the app, upload the photo, open
**🖼️ Recipe Gallery** in the left column, pick a group (Recommended, Scene /
creative, a category such as Cosplay, or All curated) and press **Preview
recipes**. Every recipe in the group renders on a small copy of your photo,
about half a second each (all 65 curated recipes took 39 s on a laptop CPU).
Click a thumbnail to see it next to the original and load that recipe; then
press Render Preview for full quality.

The same previews as one contact sheet image:

```bash
.venv/bin/python -m retouch.recipe_gallery photo.jpg -o gallery.jpg              # recommended recipes
.venv/bin/python -m retouch.recipe_gallery photo.jpg -o gallery.jpg --group all  # all 65
.venv/bin/python -m retouch.recipe_gallery photo.jpg --group cosplay --size 768
```

Previews run at 512 px on the long edge (`--size` changes it), so fine skin
texture differs a little from a full-resolution render.

### Recipe looks as .cube LUTs

Any recipe's colour look can be saved as a standard 33-point `.cube` 3D LUT, so
the same grade can go on photos in Photoshop or Lightroom and on video in
Premiere, DaVinci Resolve or Final Cut.

```bash
./run lut cosplay_feed_pop_v1 -o ~/luts           # writes ~/luts/cosplay_feed_pop_v1.cube
./run lut apex_cinema_v1 -o ~/luts/cinema.cube    # pick the file name
./run lut --all -o ~/luts                         # one file per curated recipe with a colour look
./run batch --recipe apex_cinema_v1 --export-lut ~/luts   # same thing from the batch CLI
```

In the app, pick a recipe, open **🎨 Save Look as LUT** in the left column and
press **Save .cube LUT** to download it. `--size` / `--lut-size` changes the
grid (33 is what Photoshop, Premiere and Resolve use; 65 is finer and larger).

A LUT only maps one colour to another, so it carries the recipe's per-pixel
colour and tone steps and nothing else:

- **In the LUT:** contrast, brightness, highlights/shadows/whites/blacks,
  vibrance and saturation, white balance, tonal and film curves, the colour
  grade preset, HSL and calibration, fade, highlight drift, film-look LUTs,
  negative split tone, split toning and the B&W mixer.
- **Not in the LUT (full app only):** all face and skin retouching, reshaping,
  background and subject work, and anything that looks at neighbouring
  pixels: clarity, glow and bloom, haze, vignette, grain, halation,
  chromatic aberration, sharpening and the impact finish. Steps the recipe
  applies only on skin (three-way split toning, fade toe, skin glow) are left
  out too, so the LUT matches what the recipe does to everything outside the
  face. The command lists which of these a recipe uses.

Recipes that only retouch (such as `natural`) have no colour look, so the
command says so and `--all` skips them. Across all 68 curated recipes the LUT
matches the app's own colour steps to within one 8-bit level on average;
strong curves (the `game_character` looks) differ by up to about 7 levels on
1% of pixels, the usual interpolation limit of a 33-point LUT.

### Watermark / credit on copies for posting

Reposts drop names, so a small credit on the image itself helps. The
watermark goes on **copies** in `<output>/watermarked/`; the retouched
originals stay clean. It is off until you give a credit.

```bash
./run batch ~/photos -o ~/out --recipe natural --watermark "© Alex Studio {year}"
./run batch ~/photos -o ~/out --recipe natural --watermark "@alexshoots" --social-crops
./run watermark ~/out --text "© Alex Studio" --logo logo.png --position bottom-left
```

- **Placement:** `auto` (the default) uses the bottom-right corner unless a
  face (plus room for wig and chin) is there, then tries bottom-left, top-right,
  top-left and bottom-centre. `--watermark-position` / `--position` forces a spot.
- **Look:** text size is 3% of the photo's short side (`--watermark-size`),
  70% opacity (`--watermark-opacity 0-100`), white on dark backgrounds and black
  on light ones with a soft shadow (`--watermark-color white|black` to force).
  `{year}` becomes the current year. `--watermark-logo` adds a PNG logo, with
  transparency, before the text (or on its own).
- **Social crops:** with `--social-crops` each crop gets the credit too,
  placed clear of the faces in that crop.
- **Font:** the built-in font is Aileron Regular (CC0, embedded in Pillow). It
  covers Latin letters, digits, `©`, `@` and `·` but not accented or CJK
  characters; pass `--watermark-font your.ttf` for those (a warning names any
  missing character).
- **Files:** sRGB JPEG, quality 92. Camera EXIF (GPS, serial) is not carried
  over; the credit is written to the EXIF Copyright field (`©` as `(C)`).

In the app: the **© Watermark / Credit** accordion in the left column stamps
the exported photo (or the preview, if nothing is exported yet) and gives you
the file; the Batch tab has a **Watermark / credit** box and position.

## Tests

```bash
uv sync --locked --extra desktop --extra dev   # or: pip install -r requirements/dev.txt
scripts/dev/test tests/ -v
```

## Development

Auto-reload the GUI on file changes (requires `watchfiles`):

```bash
pip install watchfiles
watchfiles "python3 gui.py" . retouch
```

This restarts the Gradio GUI whenever a file under `.` or `retouch/` changes.

## Documentation

- [API.md](docs/architecture/API.md) — Python API reference and parameter list
- [ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) — pipeline design and module breakdown
- [BATCH_GUIDE.md](docs/guides/BATCH_GUIDE.md) — batch processing examples
- [SPLIT_SHOOT.md](docs/guides/SPLIT_SHOOT.md) — split a shoot into folders by capture time
- [RECIPE_SWEEP.md](docs/RECIPE_SWEEP.md) — recipe comparisons and folder visual QA
- [GUI.md](docs/guides/GUI.md) — Gradio web UI layout, components, and styling
- [RECIPE_GUIDE.md](docs/guides/RECIPE_GUIDE.md) — recipe authoring reference
- [docs/INDEX.md](docs/INDEX.md) — full documentation map

## License

The original Retouch Engine source code is released under the [MIT License](LICENSE).

Third-party assets retain their own terms. Model sources and licenses are recorded
in [models/manifest.json](models/manifest.json), and the bundled demo LUT policy is
documented in [luts/ACQUISITION.md](luts/ACQUISITION.md). Do not commit commercial
LUTs or model files without confirming their redistribution rights.

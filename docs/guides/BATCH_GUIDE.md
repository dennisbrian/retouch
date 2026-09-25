# Batch Face Retouching Guide

This guide describes how to run the batch processing CLI tool for the **Pro Max Face Retouch Engine** to process folders of images in parallel.

---

## 1. Quick Start

To run a batch of images with the **`cosplay_character_showcase_v1`** recipe (dewy skin, vivid eyes, wig-lace blend):

```bash
python3 cli.py "/path/to/input_folder" -o "/path/to/output_folder" --recipe cosplay_character_showcase_v1
```

`--recipe` only accepts the curated recipe names. Run `./run recipes` to list
them; older looks such as `cyber_doll` or `anime_v2` are Python-API only (see
Example F).

By default, this will:
*   Search the input folder for images (`.jpg`, `.jpeg`, `.png`, `.webp`, etc.).
*   Process them in parallel using **half of your CPU cores**.
*   Save the retouched images in the output folder.
*   Generate side-by-side comparison images (e.g. `filename_compare.jpg`).

For a non-destructive preview of the exact selection and planned destinations,
add `--dry-run --input-plan /path/to/plan.json`. The preview does not create the
output directory or start the retouch engine. Any source-overwrite or collision
condition is shown as a blocking execution warning.

---

## 2. Command Reference & Common Options

| Option | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| **`--recipe`** / `--preset` | | `natural` | Choose the retouch recipe from the curated catalog, e.g. `natural`, `portrait`, `natural_polish_v1`, `cosplay_clear_v1`, `cosplay_character_showcase_v1`, `apex_cinema_v1`, `reala_ace`. Run `./run recipes` for the full list. |
| **`--impact`** | | Recipe default | Global punch/finish intensity from `0` to `100`. Useful when a result feels too weak after normal retouching. |
| **`--global-only`** | | *Off* | Skip face detection and local skin/eye/lip edits. Use this for fast color, contrast, glow, and impact retouching, or when MediaPipe cannot run in the current environment. |
| **`-o`** / `--output` | | *None* | Directory where output images and comparisons are saved. |
| **`-f`** / `--force` | | *Off* | Overwrite existing files in the output directory. |
| **`-r`** / `--recursive` | | *Off* | Recursively search subdirectories for images. |
| **`--workers`** | | `CPU cores / 2` | Number of parallel worker processes. Increase to speed up processing (e.g., `--workers 6` or `--workers 8`). |
| **`--max-dim`** | | *Original* | Downscale the longest side of the image to `N` pixels before processing (speeds up CPU processing significantly). e.g., `--max-dim 2048`. |
| **`--quality`** | `-q` | `95` | Compression quality for JPEG/WebP output (1–100). |
| **`--format`** | | `same` | Output image format: `jpg`, `png`, `webp`, or `same` to match source. |
| **`--input-list`** | | *None* | Load literal paths from a versioned JSON list; relative paths use the list's `base_dir` or directory. |
| **`--input-plan`** | | *None* | Write the deterministic input selection, planned artifacts, and per-file results as JSON. |
| **`--resume-plan`** | | *None* | Skip only rows whose settings, source hash, output path, and output hash still match; every other selected row re-renders (dry-run shows why: settings changed / source changed / output missing / output modified). A re-render replaces the plan's own prior output without `-f` only if that file still hashes to the plan's record; any other existing output blocks the run unless `-f` is given. |
| **`--include`** | | *None* | Include discovered paths matching a repeatable pattern. |
| **`--exclude`** | | *None* | Exclude discovered paths matching a repeatable pattern. |
| **`--include-hidden`** | | *Off* | Include hidden files/directories during folder discovery. |
| **`--raw-jpeg-policy`** | | `error` | Resolve matching RAW+JPEG pairs as `raw-only`, `jpeg-only`, or `suffix`; default reports the conflict. |
| **`--input-check`** | | `paths` | Use `headers` for container checks or `decode` for the configured decoder before rendering. |
| **`--max-input-pixels`** | | *None* | With header/decode checks, reject images larger than the specified pixel count. |
| **`--multi-frame-policy`** | | `error` | With header/decode checks, reject multi-frame inputs or explicitly allow first-frame processing. |
| **`--ram-budget-gib`** | | *None* | Opt-in cap on workers from estimated decoded working memory. It is an estimate, not a peak-RAM guarantee. |
| **`--skip-disk-check`** | | *Off* | Bypass the destination-volume free-space estimate (use only with an explicit operator decision). |
| **`--raf-decoder`** | | `rawpy` | RAF development path: native 16-bit `rawpy` (default), camera-JPEG `raf2jpeg`, or `rawpy-fuji-match` for full-resolution RAW calibrated to the camera preview. |
| **`--raf2jpeg-path`** | | Auto | Explicit `raf2jpeg` executable path. By default Retouch discovers the sibling `../raf2jpeg/bin/raf2jpeg` checkout, then searches `PATH`. |
| **`--raf2jpeg-quality`** | | `100` | JPEG quality passed to a `raf2jpeg` re-encoding fallback. The normal embedded-camera-JPEG path preserves its original bytes unchanged. |
| **`--fuji-match-strength`** | | `0.85` | Blend from the native RAW development (0) to the camera-preview calibration (1), used with `--raf-decoder rawpy-fuji-match`. |
| **`--no-compare`** | | *Off* | Skip generating the `_compare` side-by-side comparison files. |
| **`--no-exif`** | | *Off* | Skip copying EXIF metadata (orientation, camera tags, etc.) from the source image. |
| **`--dry-run`** | | *Off* | Print the input plan, settings, and safety conditions without creating outputs or executing retouching. |
| **`--export-lut`** | | *None* | Save `--recipe`'s colour look as a `.cube` 3D LUT and exit (no photos needed). Takes a `.cube` path or a folder; with no value it writes `<recipe>.cube` into `-o` or the current folder. Only per-pixel colour and tone steps go in; see "Recipe looks as .cube LUTs" in the README. |
| **`--lut-size`** | | `33` | Grid points per axis for `--export-lut`. |
| **`--progress-file`** | | `<output>/.retouch-progress.json` | Where to write the live progress JSON (see below). |
| **`--no-progress-file`** | | *Off* | Do not write the progress JSON. |

`--format same` preserves PNG and WebP. Other recognized source formats,
including TIFF, BMP, RAW, and EXR, currently resolve to JPEG unless an explicit
format is requested; a 16-bit request forces PNG/TIFF as documented by the CLI.

### Live progress, stopping and resuming

The progress bar measures work in megapixels and shows each active image's
current stage and completed face count. If an image remains in one stage for
90 seconds, the reporter prints a heartbeat line with the image, stage, and
elapsed time.

`<output>/.retouch-progress.json` is rewritten about once a second with the
counts, ETA, per-image status, stage, elapsed time, face count, stage timings,
and QA results. Use `--progress-file /path/to/progress.json` to choose another
location, or `--no-progress-file` to disable the file. The run summary lists
the slowest images and any failures or QA flags.

Press Ctrl-C once to stop. Queued images are cancelled, active workers are
ended, and partial atomic-write files are removed. Completed images remain
available; rerunning the same command skips existing outputs and processes the
rest. Use `-f/--force` to redo everything.

---

## 3. Practical Examples

### Recipe Comparison and Folder Visual QA

Use the dedicated QA runners when selecting a look, validating a new recipe, or
checking a folder before a production batch. Unlike `cli.py`, they retain one
output directory per recipe, per-recipe contact sheets, and a `manifest.json`.

```bash
# One image through selected recipes
./executable/recipe_sweep portrait.jpg --recipes natural,portrait,cosplay --compare

# One or more recipes across a folder. Omit --max-dim for a full-resolution run.
./executable/recipe_batch /path/to/input_folder \
  -o test_output/recipe_validation \
  --recipes cosplay_character_showcase_v1,cosplay_pastel_dream_showcase_v1 \
  --compare
```

Use `--global-only --max-dim 1600` only for a fast global-grade smoke check; it
does not exercise face detection, masks, or local retouching. See
[Recipe Sweep](../RECIPE_SWEEP.md) for output layout and all options.

### Example A: Fast Preview Batch
If you have a large folder of images (e.g. 500+ photos) and want a fast preview of the stronger reference-style look, downscale them to `2048px` and use maximum CPU threads:

```bash
python3 cli.py "/Users/dennis/Pictures/Photoshoot" \
  -o "/Users/dennis/Desktop/Processed" \
  --recipe cosplay_character_showcase_v1 \
  --global-only \
  --max-dim 2048 \
  --workers 8
```

### Example B: High-Quality Production Export
To process raw-exported full-resolution JPEGs, preserve EXIF data, keep maximum quality (`-q 98`), and output as WebP format:

```bash
python3 cli.py "/Users/dennis/Pictures/Photoshoot" \
  -o "/Users/dennis/Desktop/Final_WebP" \
  --recipe cosplay_clear_v1 \
  --format webp \
  -q 98
```

### Example C: Dry-Run Preview
Check how many images are in a folder and verify options before running:

```bash
python3 cli.py "/Users/dennis/Pictures/Photoshoot" --recipe cosplay_clear_v1 --dry-run
```

### Example D: RAF Through `raf2jpeg`

Use this when you prefer the RAF rendition produced by your local
`raf2jpeg` tool. Retouch creates the converter JPEG in a temporary directory,
retouches it, then removes the temporary file; it never writes a JPEG beside
the source RAF. The native `rawpy` decoder remains the default because it
keeps 16-bit float data for more grading headroom.

```bash
python3 cli.py "/Users/dennis/Pictures/Photoshoot/DSCF0001.RAF" \
  -o "/Users/dennis/Desktop/Retouched" \
  --raf-decoder raf2jpeg \
  --raf2jpeg-quality 100 \
  --recipe reala_ace \
  --format png
```

If `raf2jpeg` is not in its sibling checkout or on `PATH`, provide it
explicitly:

```bash
python3 cli.py DSCF0001.RAF --raf-decoder raf2jpeg \
  --raf2jpeg-path /path/to/raf2jpeg/bin/raf2jpeg -o retouched
```

### Example E: Full-Resolution RAF With Fuji-Matched Colour

This is the recommended compromise when the embedded JPEG is too small for
delivery. Retouch develops the full RAF through its 16-bit path, learns a
global tone curve and Lab colour calibration from the embedded Fuji JPEG, then
retouches the full-resolution result. It preserves RAW detail, but cannot copy
the camera's proprietary local sharpening and noise reduction pixel-for-pixel.

```bash
python3 cli.py DSCF0001.RAF -o retouched \
  --raf-decoder rawpy-fuji-match \
  --fuji-match-strength 0.85 \
  --recipe reala_ace \
  --format png
```

Use `0.65`–`0.75` for a more restrained camera match, or `1.0` for the
strongest global match.

### Example F: Scene and Creative Looks
The curated catalog includes scene and creative looks. They are tuned for
matching light or intent, so check a few results before running a whole shoot:

```bash
# Cinematic film look: Eterna base, halation, grain, lens-blur separation
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe apex_cinema_v1 --workers 6

# AAA game character: directional relight, micro-contrast
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe game_character_v1 --workers 6

# Japanese transparent skin: airy haze, faded toe
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe jp_transparent_v1 --workers 6

# Cold, icy scene
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe cosplay_ice_cathedral_v1 --workers 6

# Warm, heroic amber scene
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe cosplay_heroic_amber_v1 --workers 6
```

The older anime and dreamy recipes (`anime_cinematic_v1`, `anime_cinematic_soft`,
`anime_cinematic_action`, `anime_crystal_void`, `anime_cinematic_fantasy`,
`fuji_porcelain`, `blue_dream`, `xhs_ultrasoft`, `cyber_doll`) are still in
`retouch/recipes.py` but are not in the curated catalog, so the CLI and GUI
reject them. Run them through the Python API:

```python
from pathlib import Path
import cv2
from retouch.engine import RetouchEngine

if __name__ == "__main__":  # required: the engine starts worker processes
    out = Path("/path/to/output"); out.mkdir(parents=True, exist_ok=True)
    with RetouchEngine() as engine:
        for src in sorted(Path("/path/to/input").glob("*.jpg")):
            result = engine.process(cv2.imread(str(src)), recipe="anime_cinematic_v1")
            cv2.imwrite(str(out / src.name), result.image)
```

Save it as a `.py` file and run it with `.venv/bin/python`; the engine's face
workers cannot start from `python -` or an interactive paste.
---

## 4. Reviewing a Batch

After a batch runs, the output folder contains `review.html` — an interactive page to review all processed images in your browser.

### Opening the Review Page

Open the review page from your file manager or browser: `<output-folder>/review.html`. It shows:

- Before/after previews for each image (hold Space to flip)
- Engine QA flags (if any were raised)
- Close-up face crops (before vs after pairs)
- Input filename and processing status
- Links to full-resolution output and comparison images

All data is embedded in the HTML file; no external network or server is needed. The hidden `.retouch-review/` folder next to it holds the previews and the per-image QA records the page is built from. Deleting it frees space and never touches your photos, but the page loses its images and QA results (rebuild with `./run review build`, which then shows QA as "not recorded").

The CLI prints the page's path at the end of the batch (`Review page → …`). The app's Batch tab writes the same page and adds its path to the log.

### Navigation & Decisions

Use keyboard shortcuts to review images and mark them:

- **J** / **K** — Next / Previous image (within current filter)
- **←** / **→** — Same as J/K
- **Enter** — Open the image in detail view
- **Esc** — Back to grid
- **P** — Mark as a pick (jumps to the next image)
- **X** — Mark as rejected (jumps to the next image)
- **U** — Clear the pick/reject decision
- **F** — Cycle through filters (Shift+F goes back) (All → Flagged → Failed → Unreviewed → Picked → Rejected)
- **Space** (hold) — Show the before image while viewing after
- **C** — Toggle side-by-side before/after comparison
- **?** — Show keyboard help overlay

Your decisions are saved in this browser automatically (per batch), so you can close the page and come back. They stay in that browser only, so export them before switching machines. On touch devices, press-and-hold the image to show the before version.

### Exporting Decisions

After you've marked your picks and rejects:

1. Click the **Export** button in the header
2. A JSON file downloads, named like `3f2a9c1b04de-decisions.json`
3. The dialog shows the exact `./run review apply …` command for this batch, with a Copy button

The file contains your `pick` / `reject` decisions for the batch, keyed by image ID.

### Applying Decisions

Copy picks to a folder and optionally move rejects:

```bash
./run review apply /path/to/output decisions.json [--move-rejects]
```

This:
- **Copies** all picked images to `/path/to/output/picks/` (preserving subdirectory structure)
- **Moves** rejected images (and their comparisons) to `/path/to/output/rejected/` only if `--move-rejects` is passed
- **Never** overwrites or deletes existing files — any collisions are skipped and reported

### Building a Review Page for an Older Batch

If you have an output folder from an earlier batch run without a `review.html`, generate one now:

```bash
./run review build /path/to/output --source /path/to/input
```

For an older batch, QA metrics are not available in the page (they are shown as "not recorded").

### Skipping the Review Page

To batch-process without generating a review page, add `--no-review`:

```bash
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe cosplay_clear_v1 --no-review
```

---

## 5. Social Crops for Instagram, Reels & Stories

Automatically export face-aware crops of your retouched photos in popular social media formats. The engine detects the subject's face, centers the crop on it, preserves headroom for wigs and headpieces, and ignores small background faces. Each crop is output as sRGB JPEG (quality 92, no EXIF), downscaled to `1080px` width with light sharpening, and never upscaled.

### Supported Formats

| Format | Aspect Ratio | Output Size | Use Case |
| :--- | :---: | :--- | :--- |
| **`4:5`** | 4:5 | 1080×1350 | Instagram feed posts |
| **`9:16`** | 9:16 | 1080×1920 | Reels, TikTok, Stories |
| **`1:1`** | 1:1 | 1080×1080 | Instagram square, profile |
| **`3:4`** | 3:4 | 1080×1440 | Instagram profile grid, Threads |

### Method 1: During Batch Processing

Add the `--social-crops` flag when running a batch to export crops alongside the retouched images:

```bash
# Export default formats (4:5, 9:16, 1:1)
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe natural --social-crops

# Export specific formats
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe natural --social-crops 4:5,9:16

# Export all formats
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe natural --social-crops all
```

Use the launcher for simplicity:

```bash
./run batch ~/photos -o ~/photos_out --recipe natural --social-crops
```

Crops are written to `<output>/social/<4x5|9x16|1x1|3x4>/<name>_<slug>.jpg`. Existing crops are skipped unless you pass `--force`. Add `--social-size full` to keep the native crop resolution instead of 1080 px wide.

The top of the crop follows the person mask above the face, so tall wigs, ears and headpieces stay in frame whenever the shape allows it.

### Method 2: On Already-Retouched Folders

Process a folder of retouched images without re-running the full batch pipeline:

```bash
python3 -m retouch.social_crops ~/photos_out
```

Or use the launcher:

```bash
./run crops ~/photos_out
```

**Options:**

| Option | Default | Description |
| :--- | :--- | :--- |
| `--formats` | `4:5,9:16,1:1` | Comma-separated list: `4:5`, `9:16`, `1:1`, `3:4`, or `all`. |
| `--size` | `platform` | `platform` outputs at `1080px` width; `full` keeps native crop resolution. |
| `--quality` | `92` | JPEG quality (1–100). |
| `--force` | *Off* | Overwrite existing crop files. |
| `-o` / `--output` | `<input>/social` | Output directory for the crop folder structure. |
| `-r` / `--recursive` | *Off* | Recursively process subdirectories. |

**Example:**

```bash
# Full-resolution crops with custom quality
python3 -m retouch.social_crops ~/photos_out \
  --formats 9:16,1:1 \
  --size full \
  --quality 95

# Recursively process all subfolders
python3 -m retouch.social_crops ~/batch_output \
  -r --force --formats all
```

### Method 3: GUI Batch Tab

When using the batch interface in the GUI, check the "Social crops" checkbox and select your desired formats (4:5, 9:16, 1:1, 3:4). Crops are exported to `<output>/social/` after the batch completes.

### Behavior Details

- **No faces detected:** Falls back to center-crop of the frame.
- **Multiple faces:** Crops are centered on the largest face or group of similar-size faces; small background faces are ignored.
- **EXIF stripped:** Geolocation and camera metadata are removed for privacy when posting.
- **Skip rules:** Comparison images (`*_compare.*`) and anything already inside a `social/` folder are skipped.
- **Platform downsample:** Output is always downscaled to `1080px` width (unless `--size full` is set) to match platform delivery specs, with light output sharpening applied.

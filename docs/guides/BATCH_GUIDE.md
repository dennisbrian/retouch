# Batch Face Retouching Guide

This guide describes how to run the batch processing CLI tool for the **Pro Max Face Retouch Engine** to process folders of images in parallel.

---

## 1. Quick Start

To run a batch of images with the stronger **`cyber_doll`** recipe for the pink/cyan high-impact cosplay look:

```bash
python3 cli.py "/path/to/input_folder" -o "/path/to/output_folder" --recipe cyber_doll --impact 70 --global-only
```

By default, this will:
*   Search the input folder for images (`.jpg`, `.jpeg`, `.png`, `.webp`, etc.).
*   Process them in parallel using **half of your CPU cores**.
*   Save the retouched images in the output folder.
*   Generate side-by-side comparison images (e.g. `filename_compare.jpg`).

---

## 2. Command Reference & Common Options

| Option | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| **`--preset`** / `--recipe` | | `natural` | Choose the retouch recipe: `natural`, `portrait`, `beauty`, `cosplay`, `cosplay_3d`, `cosplay_no_eq`, `cyber_doll`, `scifi_cosplay`, `anime_cosplay`, `anime_cinematic_v1`, `anime_cinematic_soft`, `anime_cinematic_action`, `anime_crystal_void`, `anime_cinematic_fantasy`, `xiaohongshu`, `xhs_ultrasoft`, `pink_dream`, `blue_dream`, `fantasy_goddess`, `fuji_porcelain`, `dreamy`, `magazine`, `korean_beauty`, `idol`, `wedding`. |
| **`--impact`** | | Recipe default | Global punch/finish intensity from `0` to `100`. Useful when a result feels too weak after normal retouching. |
| **`--global-only`** | | *Off* | Skip face detection and local skin/eye/lip edits. Use this for fast color, contrast, glow, and impact retouching, or when MediaPipe cannot run in the current environment. |
| **`-o`** / `--output` | | *None* | Directory where output images and comparisons are saved. |
| **`-f`** / `--force` | | *Off* | Overwrite existing files in the output directory. |
| **`-r`** / `--recursive` | | *Off* | Recursively search subdirectories for images. |
| **`--workers`** | | `CPU cores / 2` | Number of parallel worker processes. Increase to speed up processing (e.g., `--workers 6` or `--workers 8`). |
| **`--max-dim`** | | *Original* | Downscale the longest side of the image to `N` pixels before processing (speeds up CPU processing significantly). e.g., `--max-dim 2048`. |
| **`--quality`** | `-q` | `95` | Compression quality for JPEG/WebP output (1–100). |
| **`--format`** | | `same` | Output image format: `jpg`, `png`, `webp`, or `same` to match source. |
| **`--raf-decoder`** | | `rawpy` | RAF development path: native 16-bit `rawpy` (default), camera-JPEG `raf2jpeg`, or `rawpy-fuji-match` for full-resolution RAW calibrated to the camera preview. |
| **`--raf2jpeg-path`** | | Auto | Explicit `raf2jpeg` executable path. By default Retouch discovers the sibling `../raf2jpeg/bin/raf2jpeg` checkout, then searches `PATH`. |
| **`--raf2jpeg-quality`** | | `100` | JPEG quality passed to a `raf2jpeg` re-encoding fallback. The normal embedded-camera-JPEG path preserves its original bytes unchanged. |
| **`--fuji-match-strength`** | | `0.85` | Blend from the native RAW development (0) to the camera-preview calibration (1), used with `--raf-decoder rawpy-fuji-match`. |
| **`--no-compare`** | | *Off* | Skip generating the `_compare` side-by-side comparison files. |
| **`--no-exif`** | | *Off* | Skip copying EXIF metadata (orientation, camera tags, etc.) from the source image. |
| **`--dry-run`** | | *Off* | Scan the directories and print settings without executing any retouching. |

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
  --recipe cyber_doll \
  --impact 70 \
  --global-only \
  --max-dim 2048 \
  --workers 8
```

### Example B: High-Quality Production Export
To process raw-exported full-resolution JPEGs, preserve EXIF data, keep maximum quality (`-q 98`), and output as WebP format:

```bash
python3 cli.py "/Users/dennis/Pictures/Photoshoot" \
  -o "/Users/dennis/Desktop/Final_WebP" \
  --preset cosplay_3d \
  --format webp \
  -q 98
```

### Example C: Dry-Run Preview
Check how many images are in a folder and verify options before running:

```bash
python3 cli.py "/Users/dennis/Pictures/Photoshoot" --preset cosplay_3d --dry-run
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
  --recipe fuji_porcelain \
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
  --recipe fuji_porcelain \
  --format png
```

Use `0.65`–`0.75` for a more restrained camera match, or `1.0` for the
strongest global match.

### Example F: Anime Cinematic / Dreamy Recipes
The engine ships with several specialised anime and dreamy recipes that work
well on illustration-style or soft-light photoshoots. Pick one with
`--recipe`/`--preset`:

```bash
# Bright, glossy anime key visual look
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe anime_cinematic_v1 --workers 6

# Soft, low-contrast anime variant
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe anime_cinematic_soft --workers 6

# High-contrast action shot
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe anime_cinematic_action --workers 6

# Crystal-clear negative-space look
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe anime_crystal_void --workers 6

# Fantasy anime palette
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe anime_cinematic_fantasy --workers 6

# Soft Fuji-style porcelain skin finish
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe fuji_porcelain --workers 6

# Cool blue dreamy palette
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe blue_dream --workers 6

# Xiaohongshu ultra-soft beauty look
python3 cli.py "/path/to/input" -o "/path/to/output" --recipe xhs_ultrasoft --workers 6
```

---

## 4. Social Crops for Instagram, Reels & Stories

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

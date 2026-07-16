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

### Example D: Anime Cinematic / Dreamy Recipes
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

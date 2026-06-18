# Batch Face Retouching Guide

This guide describes how to run the batch processing CLI tool for the **Pro Max Face Retouch Engine** to process folders of images in parallel.

---

## 1. Quick Start

To run a batch of images with the new **`cosplay_3d`** recipe (which preserves natural 3D highlights and shading):

```bash
python3 cli.py "/path/to/input_folder" -o "/path/to/output_folder" --preset cosplay_3d
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
| **`--preset`** / `--recipe` | | `natural` | Choose the retouch recipe: `cosplay_3d`, `cosplay_no_eq`, `cosplay`, `natural`, `portrait`, `beauty`, `anime`, `xiaohongshu`, `dreamy`, `magazine`. |
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

### Example A: Fast Preview Batch
If you have a large folder of images (e.g. 500+ photos) and want a fast preview of the `cosplay_3d` look, downscale them to `2048px` and use maximum CPU threads:

```bash
python3 cli.py "/Users/dennis/Pictures/Photoshoot" \
  -o "/Users/dennis/Desktop/Processed" \
  --preset cosplay_3d \
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

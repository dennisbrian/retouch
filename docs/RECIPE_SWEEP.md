# Recipe Sweep — visual diff across all recipes

Renders one input through every registered retouch recipe. Exports one output per recipe plus a contact sheet and QA manifest.

## Quick start

```bash
./executable/recipe_sweep photo.jpg
```

Output goes to `test_output/recipe_sweeps/<stem>_<timestamp>/`.

## Usage

```
./executable/recipe_sweep INPUT_IMAGE [OUTPUT_DIR] [options]
```

| Argument | Description |
|----------|-------------|
| `INPUT_IMAGE` | Path to photo (required) |
| `OUTPUT_DIR` | Optional output folder (default: `test_output/recipe_sweeps/...`) |

If `OUTPUT_DIR` is omitted, a timestamped folder is created automatically.
Pass `-o DIR` or `--output DIR` instead of the positional arg if preferred.

## Options

All flags after the first two args are forwarded to `scripts/recipes/recipe_sweep.py`:

| Flag | Description |
|------|-------------|
| `--recipes natural,portrait` | Comma-separated subset (default: all) |
| `--compare` | Side-by-side input vs output |
| `--global-only` | Skip face detection, global grade only |
| `--max-dim 1600` | Downscale before processing (faster) |
| `--format png` | Output format (jpg/png, default: jpg) |
| `--quality 95` | JPEG quality (default: 92) |
| `--fail-on-qa` | Exit 1 if any QA detector flags |

## Examples

```bash
# Quick check on 2 recipes at proxy res
./executable/recipe_sweep photo.jpg --recipes natural,portrait --max-dim 1600

# Full-res PNG, side-by-side compare
./executable/recipe_sweep photo.jpg test_output/recipe_sweeps/my_check --compare --format png

# Global-only (no MediaPipe models needed)
./executable/recipe_sweep photo.jpg --global-only --max-dim 1600
```

## Output layout

```
test_output/recipe_sweeps/<stem>_<timestamp>/
├── contact_sheet.jpg        # All recipes in one grid
├── manifest.json            # Recipes, timings, QA scores
├── <recipe1>.jpg
├── <recipe2>.jpg
└── ...
```

Each per-recipe output has metadata (timings, QA warnings) printed to stdout during processing. The `manifest.json` records everything for programmatic review.

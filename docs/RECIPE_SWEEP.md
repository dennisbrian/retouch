# Recipe Sweep — visual diff across maintained recipes

Renders one input through retouch recipes and exports one output per recipe plus a contact sheet and QA manifest. The default is the correction-first `recommended` set. The 50-recipe catalog is available with `--recipes curated`; venue and cinematic looks are deliberately labelled conditional rather than treated as universal edits.

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
| `--recipes natural,portrait` | Comma-separated subset. Use `recommended` (default), `conditional`, `curated` (all 50 maintained), or `all` (legacy/experimental). |
| `--compare` | Side-by-side input vs output |
| `--global-only` | Skip face detection, global grade only |
| `--max-dim 1600` | Downscale before processing (faster) |
| `--format png` | Output format (jpg/png, default: jpg) |
| `--quality 95` | JPEG quality (default: 92) |
| `--fail-on-qa` | Exit 1 if any QA detector flags |
| `--resume` | Resume a partial sweep, skipping recipes already recorded as done |
| `--restart-engine-per-recipe` | Recreate native engine per recipe; slower, but reduces memory pressure |

## Examples

```bash
# Quick check on 2 recipes at proxy res
./executable/recipe_sweep photo.jpg --recipes natural,portrait --max-dim 1600

# Full curated review. These 50 include scene-specific and creative looks,
# so judge them only against a matching source photo.
./executable/recipe_sweep photo.jpg --recipes curated --compare --max-dim 1600

# Full-res PNG, side-by-side compare
./executable/recipe_sweep photo.jpg test_output/recipe_sweeps/my_check --compare --format png

# Global-only (no MediaPipe models needed)
./executable/recipe_sweep photo.jpg --global-only --max-dim 1600

# Long full-pipeline QA: proxy resolution, checkpointed, and memory-safe.
# Re-run the same command with --resume after a native failure or interruption.
./executable/recipe_sweep photo.jpg test_output/recipe_sweeps/full_qa \
  --recipes all --compare --max-dim 1600 --restart-engine-per-recipe

./executable/recipe_sweep photo.jpg test_output/recipe_sweeps/full_qa \
  --recipes all --compare --max-dim 1600 --restart-engine-per-recipe --resume
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

## Folder Batch QA

Use the batch runner to apply one or more recipes across a whole folder. It writes
one folder and contact sheet per recipe, plus a batch-level `manifest.json`.

```bash
# Reliable visual-QA pass when local MediaPipe is unavailable
./executable/recipe_batch ~/Desktop/duotiannikke \
  -o test_output/visual_qa_duotiannikke/character_batch \
  --recipes cosplay_character_showcase_v1,cosplay_heroic_amber_showcase_v1 \
  --global-only --compare --max-dim 1600

# Full face pipeline when MediaPipe is available
./executable/recipe_batch ~/Desktop/duotiannikke \
  -o test_output/visual_qa_duotiannikke/character_batch_full \
  --recipes cosplay_character_showcase_v1 --compare
```

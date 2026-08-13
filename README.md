# Pro Max Face Retouch Engine

[![Tests](https://github.com/dennisbrian/retouch/actions/workflows/test.yml/badge.svg)](https://github.com/dennisbrian/retouch/actions/workflows/test.yml)
[![Benchmarks](https://github.com/dennisbrian/retouch/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/dennisbrian/retouch/actions/workflows/benchmarks.yml)

Professional automated face retouching for portraits, cosplay, and batch workflows. Combines MediaPipe landmarks, BiSeNet semantic segmentation, frequency-separation skin work, and global color grading.

## Requirements

- Python 3.9+
- macOS, Linux, or Windows

## Install

```bash
pip install -r requirements/base.txt
```

Optional extras:

```bash
pip install -r requirements/dev.txt   # pytest
pip install -r requirements/gui.txt   # Gradio web UI
pip install -r requirements/raw.txt   # RAW camera file support
pip install retinaface                # improved face detection (optional)
```

## Model files

Create a `models/` directory and download these files before running the full pipeline:

```bash
mkdir -p models

# MediaPipe face landmarker (downloaded on demand; verify before use)
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task?generation=1683136941916318"

# MediaPipe selfie segmenter (downloaded on demand for hair/person masking)
curl -L -o models/selfie_segmenter.tflite \
  "https://storage.googleapis.com/download/storage/v1/b/mediapipe-models/o/image_segmenter%2Fselfie_segmenter%2Ffloat16%2Flatest%2Fselfie_segmenter.tflite?alt=media&generation=1683436453600523"

# MediaPipe Pose Landmarker Full (downloaded on demand for body reshape)
curl -L -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task?generation=1682642785209422"

# Verify every downloaded artifact against models/manifest.json.
python scripts/qa/verify_models.py \
  --model face_landmarker --model selfie_segmenter --model pose_landmarker_full

# Optional local BiSeNet face parsing ONNX (not distributed by the wheel)
# Place a manifest-matching resnet18.onnx export at:
#   models/resnet18.onnx
```

The engine falls back to landmark-based masks if `resnet18.onnx` is missing.
The manifest reports this model as unavailable until a verified release URL is
provided; local binaries are accepted only when their SHA-256 and size match.

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
python3 cli.py /path/to/photos -o /path/to/output --recipe cosplay --workers 4
```

See [BATCH_GUIDE.md](docs/guides/BATCH_GUIDE.md) for full CLI options.

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
pip install -r requirements/gui.txt
python3 gui.py
```

Opens at `http://127.0.0.1:7860`. Restart after updating:

```bash
pkill -f "python3 -u gui.py" && python3 gui.py
```

## Recipes

Built-in recipes include `natural`, `portrait`, `cosplay`, `cyber_doll`, `pink_dream`, `meitu_clone`, and others. List all names:

```bash
python3 cli.py --help
```

## Tests

```bash
pip install -r requirements/dev.txt
scripts/dev/test tests/ -v
```

## Development

Auto-reload on file changes (requires `watchfiles`):

```bash
pip install watchfiles
./dev.sh
```

The dev server watches `.` and `retouch/` for `.py` changes and restarts the Gradio GUI automatically.

## Documentation

- [API.md](docs/architecture/API.md) — Python API reference and parameter list
- [ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) — pipeline design and module breakdown
- [BATCH_GUIDE.md](docs/guides/BATCH_GUIDE.md) — batch processing examples
- [RECIPE_SWEEP.md](docs/RECIPE_SWEEP.md) — recipe comparisons and folder visual QA
- [GUI.md](docs/guides/GUI.md) — Gradio web UI layout, components, and styling
- [RECIPE_GUIDE.md](docs/guides/RECIPE_GUIDE.md) — recipe authoring reference
- [docs/INDEX.md](docs/INDEX.md) — full documentation map

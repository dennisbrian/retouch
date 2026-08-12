# Requirements Management

## Main Requirements Files

### `base.txt` (Core Dependencies)
Production dependencies for the retouching engine.
```bash
pip install -r requirements/base.txt
```

### `gui.txt` (GUI Extra)
Additional dependencies for Gradio web interface.
```bash
pip install -r requirements/gui.txt
```

### `dev.txt` (Development)
Testing, linting, and development tools.
```bash
pip install -r requirements/dev.txt
```

### `raw.txt` (Raw File Support)
Optional: RAW image format processing.
```bash
pip install -r requirements/raw.txt
```

## Installation

### Recommended: All-in-one
```bash
pip install -r requirements/base.txt -r requirements/gui.txt -r requirements/dev.txt
```

### Quick Start (Engine Only)
```bash
pip install -r requirements/base.txt
```

`opencv-contrib-python` is the supported OpenCV distribution. Do not install
`opencv-python` or `opencv-python-headless` in the same environment: the three
packages share the `cv2` module and can overwrite one another. The base
requirements also keep OpenCV below 4.12 so it remains compatible with the
NumPy 1.x ABI used by the current MediaPipe stack.

### Full Production (with GUI)
```bash
pip install -r requirements/base.txt -r requirements/gui.txt
```

## Key Dependencies
- **OpenCV** — Image processing
- **MediaPipe** — Face detection & landmarks
- **ONNX Runtime** — Model inference
- **Gradio** — Web UI (optional)
- **PyTest** — Testing framework (dev)

See individual files for full version pinning.

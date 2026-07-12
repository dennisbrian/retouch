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

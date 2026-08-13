# Requirements Management

## Main Requirements Files

### `base.txt` (Core Dependencies)
Production dependencies for the retouching engine.
```bash
pip install -r requirements/base.txt
```

### `gui.txt` (GUI Extra)
Additional dependencies for the Gradio browser interface.
```bash
pip install -r requirements/gui.txt
```

### `desktop.txt` (Native Desktop Extra)
Native pywebview shell and PyInstaller build tooling. On macOS/Python 3.9 this
also constrains PyObjC below 12 because the yanked 12.0 release incorrectly
advertised Python 3.9 support.
```bash
pip install -r requirements/desktop.txt
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

For the native desktop shell and application build tools:

```bash
pip install -r requirements/desktop.txt
```

## Key Dependencies

`jsonschema` is a runtime dependency used to validate recipe documents before
they are applied.

- **OpenCV** — Image processing
- **MediaPipe** — Face detection & landmarks

### Face-aware macOS runtime

Use the pinned MediaPipe 0.10.5 CPU/XNNPACK runtime for face-aware work. The
newer macOS wheels can require an unavailable OpenGL service before Python can
catch the failure. A clean environment is recommended because TensorFlow 2.20
and MediaPipe 0.10.5 require incompatible protobuf ranges:

```bash
python3 -m venv .venv-face-aware
.venv-face-aware/bin/python -m pip install -r requirements/dev.txt
```

Then verify the environment with:

```bash
RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
python scripts/qa/face_aware_runtime_probe.py test_output/DSCF8007.jpg
```

The probe must report `status: "pass"` and at least 468 landmarks before
running face-aware certification.

Run tests through the project wrapper so user-level TensorFlow, MediaPipe, or
protobuf packages cannot leak into the environment:

```bash
RETOUCH_PYTHON=.venv-face-aware/bin/python scripts/dev/test -q
```

Capture fidelity is an opt-in dependency: install `lensfunpy` separately when
Lensfun distortion/TCA/vignetting correction and its local database are
available. The application reports that correction as unavailable when the
optional dependency is absent; the creative chromatic-aberration and vignette
controls are not presented as optical correction.
- **ONNX Runtime** — Model inference
- **Gradio** — Web UI (optional)
- **PyTest** — Testing framework (dev)

See individual files for full version pinning.

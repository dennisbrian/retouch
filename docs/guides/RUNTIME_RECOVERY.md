# Runtime recovery and Python 3.12 migration

**Audit date:** 2026-08-15

**Scope:** dependency declarations, isolated-environment recovery, and the
separate migration track. This guide does not change application Python
modules or claim that a GUI launch is face-aware.

## Decision

Retouch has two deliberately separate runtime tracks:

| Track | Interpreter | Dependency source | Face backend | Status |
|---|---|---|---|---|
| Legacy recovery | CPython 3.9.6 for the immediate recovery target | `pyproject.toml` + locked `uv.lock` | MediaPipe 0.10.5 legacy CPU `mp.solutions.face_mesh` | The current application track; must be installed in isolation |
| Python 3.12 migration | CPython 3.12 | `requirements-py312-migration.txt`, then a separately generated migration lock | MediaPipe Tasks `mp.tasks.vision.FaceLandmarker` | Candidate only; the current application modules are not certified against it |

The repository does not currently carry a Python patch-version declaration.
The 3.9.6 choice above is the exact recovery interpreter for the known
working direction, not a claim that every Python 3.9 patch release is
equivalent.

Do not use the Python 3.12 candidate file with `uv sync`. The project lock is
the legacy lock and intentionally remains unchanged until the Tasks/API port
has its own verified resolution and acceptance evidence.

## Runtime Doctor

Run the doctor before interpreting a face-detection failure. Its default mode
is side-effect-free: it checks declared and installed versions, `pip check`,
duplicate OpenCV distributions, model hashes and controlled URLs, parser and
ONNX provider availability, and the effective Face-aware/Global-only mode.
It never downloads models.

```bash
PYTHONNOUSERSITE=1 RETOUCH_CACHE_DIR=/private/tmp/retouch-runtime-cache \
  python3 -m retouch.runtime_doctor --compact
```

`--probe-detector` is an explicit native initialization check. Run it only in
the isolated legacy environment and a GUI-attached/native session; a native
MediaPipe abort can terminate the subprocess before Python can catch it.

```bash
PYTHONNOUSERSITE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
  "$LEGACY_ENV/bin/python" -s -m retouch.runtime_doctor --probe-detector
```

The report's static provider list is not proof that a model session used that
provider. Active provider claims require a real session or render evidence.

## Legacy recovery contract

The active project declaration is the contract in `pyproject.toml`:

```text
mediapipe==0.10.5
protobuf>=3.20,<4
opencv-contrib-python>=4.8.0,<4.12
numpy>=1.24.0,<2
```

For the Python 3.9 resolution recorded in `uv.lock`, the relevant resolved
versions are:

```text
mediapipe              0.10.5
protobuf               3.20.3
opencv-contrib-python  4.11.0.86
numpy                  1.26.4
onnxruntime            1.20.1
gradio                 4.44.1  # GUI/dev extra
```

The lock contains MediaPipe 0.10.5 wheels for CPython 3.9, 3.10, and 3.11,
but no CPython 3.12 wheel. Therefore `requires-python = ">=3.9"` in the
project metadata must not be read as proof that the legacy lock installs on
Python 3.12. Python 3.12 belongs to the migration track below.

### Isolated recovery commands

These commands keep the environment, bytecode, package caches, and model
cache outside the checkout. They also prevent user-site packages—especially a
different TensorFlow/protobuf installation—from contaminating the pinned
runtime.

```bash
cd /Applications/htdocs/retouch

export RETOUCH_RECOVERY_ROOT=/private/tmp/retouch-runtime-recovery
export LEGACY_ENV="$RETOUCH_RECOVERY_ROOT/venv-legacy"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$RETOUCH_RECOVERY_ROOT/pycache"
export UV_CACHE_DIR="$RETOUCH_RECOVERY_ROOT/uv-cache"
export XDG_CACHE_HOME="$RETOUCH_RECOVERY_ROOT/xdg-cache"
export MPLCONFIGDIR="$RETOUCH_RECOVERY_ROOT/matplotlib"
export RETOUCH_CACHE_DIR="$RETOUCH_RECOVERY_ROOT/retouch-cache"
mkdir -p "$RETOUCH_RECOVERY_ROOT" "$PYTHONPYCACHEPREFIX" "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$RETOUCH_CACHE_DIR"

# Downloads CPython 3.9.6 only when it is not already managed by uv.
uv python install 3.9.6
uv venv --python 3.9.6 "$LEGACY_ENV"

# --locked is required: do not re-resolve the legacy project during recovery.
UV_PROJECT_ENVIRONMENT="$LEGACY_ENV" uv sync --locked --extra gui --extra dev

# Use the environment's interpreter explicitly; do not fall through to a
# globally installed `python3` or user-site package set.
PYTHONNOUSERSITE=1 "$LEGACY_ENV/bin/python" -s -m pip check
```

For a desktop packaging environment, use the same commands with
`--extra desktop` (and add `--extra dev` only when the test tools are needed).
The packaging script also prefers `uv sync --locked`; its pip fallback points
to `requirements/desktop.txt`, which is not currently tracked in this
checkout. That fallback is not an alternative recovery source of truth.

### Version and duplicate-package gate

Run this before interpreting any detector failure. It checks the exact
legacy versions and rejects the common duplicate-OpenCV installation that
causes NumPy requirement conflicts.

```bash
PYTHONNOUSERSITE=1 "$LEGACY_ENV/bin/python" -s - <<'PY'
from importlib import metadata
from packaging.version import Version

expected = {
    "mediapipe": "0.10.5",
    "protobuf": "3.20.3",
    "opencv-contrib-python": "4.11.0.86",
    "numpy": "1.26.4",
}
for name, wanted in expected.items():
    actual = metadata.version(name)
    if actual != wanted:
        raise SystemExit(f"{name}: expected {wanted}, got {actual}")

opencv_dists = sorted(
    dist.metadata["Name"].lower()
    for dist in metadata.distributions()
    if (dist.metadata["Name"] or "").lower().startswith("opencv-")
)
if opencv_dists != ["opencv-contrib-python"]:
    raise SystemExit(f"duplicate or unexpected OpenCV distributions: {opencv_dists}")

if not Version(metadata.version("protobuf")) < Version("4"):
    raise SystemExit("protobuf is outside the legacy <4 contract")
if not Version(metadata.version("numpy")) < Version("2"):
    raise SystemExit("NumPy is outside the legacy <2 contract")

print("legacy dependency gate: pass")
PY
```

`pip check` must pass in this isolated environment. A passing check in the
global interpreter is not sufficient.

### Face-aware and render acceptance

Use a consented real portrait at `PORTRAIT`; the repository does not assume
that a checked-in test image is appropriate for certification.

```bash
export PORTRAIT=/absolute/path/to/consented-portrait.jpg

PYTHONNOUSERSITE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
  "$LEGACY_ENV/bin/python" -s scripts/qa/face_aware_runtime_probe.py "$PORTRAIT"

export LEGACY_EVIDENCE="$RETOUCH_RECOVERY_ROOT/legacy-face-aware-evidence"
rm -rf "$LEGACY_EVIDENCE"
PYTHONNOUSERSITE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
  "$LEGACY_ENV/bin/python" -s scripts/qa/advanced_retouch_visual_qa.py \
  "$PORTRAIT" --output "$LEGACY_EVIDENCE"

PYTHONNOUSERSITE=1 "$LEGACY_ENV/bin/python" -s - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(
    (Path(os.environ["LEGACY_EVIDENCE"]) / "manifest.json")
    .read_text(encoding="utf-8")
)
required = {
    "global_only": False,
    "face_aware_run": True,
    "automatic_pass": True,
}
for key, wanted in required.items():
    if manifest.get(key) != wanted:
        raise SystemExit(f"manifest {key}={manifest.get(key)!r}, expected {wanted!r}")
print("face-aware evidence gate: pass")
PY
```

The runtime probe must report `status: "pass"`, `backend: "legacy_cpu"`,
and at least 468 landmarks per detected face. The render must create a
non-empty output/contact sheet and a manifest with `global_only: false`,
`face_aware_run: true`, and `automatic_pass: true`. The runner intentionally
leaves `human_review` separate; these checks are not final visual
certification.

If a model-backed Tasks path is exercised, verify only the models required by
that path and keep the manifest as the authority for availability and hashes:

```bash
PYTHONNOUSERSITE=1 "$LEGACY_ENV/bin/python" -s scripts/qa/verify_models.py \
  --model face_landmarker \
  --model selfie_segmenter
```

Entries in `models/manifest.json` with `availability: "unavailable"` and no
controlled release URL are not clean-install dependencies. They must remain
reported as unavailable until a release artifact and URL are supplied.

## Python 3.12 migration track

`requirements-py312-migration.txt` is a candidate input, not a lock and not a
claim that the current GUI or detector is available on Python 3.12. It moves
the package families forward, uses the MediaPipe Tasks API, and deliberately
does not carry the legacy `protobuf<4` constraint. Its modern protobuf range
is only a candidate; the resolver must still confirm a metadata-compatible
MediaPipe and ONNX Runtime combination.

Create the migration environment outside the project and compile a separate,
hash-bearing lock. The compile step is the availability check; if the index,
platform, or wheel set cannot satisfy the input, stop and record the failure
instead of weakening the constraints or modifying `uv.lock`.

```bash
cd /Applications/htdocs/retouch

export RETOUCH_RECOVERY_ROOT=/private/tmp/retouch-runtime-recovery
export MIGRATION_ENV="$RETOUCH_RECOVERY_ROOT/venv-py312-migration"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$RETOUCH_RECOVERY_ROOT/pycache-py312"
export UV_CACHE_DIR="$RETOUCH_RECOVERY_ROOT/uv-cache-py312"
mkdir -p "$RETOUCH_RECOVERY_ROOT" "$PYTHONPYCACHEPREFIX" "$UV_CACHE_DIR"

uv python install 3.12
uv venv --python 3.12 "$MIGRATION_ENV"

export MIGRATION_LOCK="$RETOUCH_RECOVERY_ROOT/requirements-py312-migration.lock.txt"
uv pip compile requirements-py312-migration.txt \
  --python-version 3.12 \
  --python-platform macos \
  --generate-hashes \
  --output-file "$MIGRATION_LOCK"
uv pip sync --python "$MIGRATION_ENV/bin/python" "$MIGRATION_LOCK"

PYTHONNOUSERSITE=1 "$MIGRATION_ENV/bin/python" -s -m pip check
PYTHONNOUSERSITE=1 "$MIGRATION_ENV/bin/python" -s - <<'PY'
import mediapipe as mp
import onnxruntime as ort

assert hasattr(mp, "tasks")
assert hasattr(mp.tasks, "vision")
assert hasattr(mp.tasks.vision, "FaceLandmarker")
print("Tasks API: available")
print("ONNX execution providers:", ort.get_available_providers())
PY
```

This only proves that the candidate packages resolve, import, and expose the
Tasks symbol. It does not prove native macOS construction, model loading,
current `retouch/detection.py` compatibility, GUI behavior, or output
quality. In particular, do not call the current legacy runtime probe a
successful migration result: the current application code contains a
version-specific macOS guard for MediaPipe 0.10.35, and the migration needs a
separate API-port test seam.

Migration acceptance requires all of the following after that port exists:

1. The generated migration lock is checked in as a separate, reviewed
   artifact; `pip check` is clean in a fresh Python 3.12 environment.
2. `mp.tasks.vision.FaceLandmarker` constructs with the verified local
   `face_landmarker.task` model and closes cleanly on the target OS.
3. ONNX providers are printed and the selected provider is verified for each
   model; no provider is assumed merely because it is installed.
4. A real portrait probe and render complete with `global_only: false`, a
   non-empty output/contact sheet, and `face_aware_run: true` plus
   `automatic_pass: true` in the manifest.
5. Human review remains a separate gate before a certification or release
   claim.

## What the current audit does and does not prove

Verified in the repository on the audit date:

- `pyproject.toml` declares the legacy MediaPipe/protobuf/OpenCV/NumPy
  bounds above.
- `uv.lock` resolves the legacy Python 3.9 package set above and has no
  MediaPipe 0.10.5 CPython 3.12 wheel entry.
- The build wrapper prefers `uv sync --locked --extra desktop`.
- No tracked `requirements*.txt` files existed before this migration input;
  older docs and the build fallback still reference untracked
  `requirements/...` paths.

Not verified by this documentation slice:

- A clean-install download of either environment, because that requires
  reachable package/model indexes and platform-specific wheels.
- `FaceDetector.available=True`, a real portrait render, or a
  `global_only: false` manifest in the current contaminated interpreter.
- Native MediaPipe Tasks startup on macOS, Gradio 5 API compatibility, ONNX
  provider correctness, or human visual review.
- Distribution of models whose manifest entries have no controlled URL.

For the migration design, the authoritative references are the [MediaPipe
Face Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python),
[MediaPipe package metadata](https://pypi.org/project/mediapipe/), the [Gradio
changelog](https://www.gradio.app/changelog), and the [ONNX Runtime install
matrix](https://onnxruntime.ai/docs/install/).
